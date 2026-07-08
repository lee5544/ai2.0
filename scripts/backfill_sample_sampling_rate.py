#!/usr/bin/env python3
"""Backfill missing samples.sampling_rate from TDMS metadata.

The script reads rows where ``label_records.db.samples.sampling_rate`` is NULL,
locates the corresponding TDMS file from the sample row or tdms_manifest.csv,
reads only TDMS metadata, and computes sampling_rate from channel
``wf_increment``.

Dry-run is the default. Use ``--execute`` to update the database.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.line_rules import LINE_RULES
from data_manager.sample_generate import SampleGenerator
from data_manager.tdms_read import open_tdms


@dataclass(frozen=True)
class ManifestRow:
    line: str
    sn: str
    reference: str
    time: str
    tdms_storage_root: str
    relative_path: str
    tdms_path: str
    created_time: str


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _read_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cfg = _read_yaml(args.config)
    database = cfg.get("database") if isinstance(cfg.get("database"), dict) else {}
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    data_root = Path(
        _text(args.data_root)
        or _text(cfg.get("data_root"))
        or _text(dataset.get("data_root"))
        or "."
    ).expanduser()
    db_path = Path(
        _text(args.database)
        or _text(database.get("label_records_db_path"))
        or _text(cfg.get("label_records_db_path"))
        or _text(dataset.get("label_records_db_path"))
        or str(data_root / "metadata" / "label_records.db")
    ).expanduser()
    manifest_path = Path(
        _text(args.manifest)
        or _text(database.get("manifest_path"))
        or _text(dataset.get("manifest_path"))
        or str(db_path.parent / "tdms_manifest.csv")
    ).expanduser()
    return data_root, db_path, manifest_path


def _load_manifest(path: Path) -> dict[tuple[str, str], list[ManifestRow]]:
    rows_by_key: dict[tuple[str, str], list[ManifestRow]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item = ManifestRow(
                line=_text(row.get("line")),
                sn=_text(row.get("sn")),
                reference=_text(row.get("reference")),
                time=_text(row.get("time")),
                tdms_storage_root=_text(row.get("tdms_storage_root")),
                relative_path=_text(row.get("relative_path")),
                tdms_path=_text(row.get("tdms_path")),
                created_time=_text(row.get("created_time")),
            )
            if item.line and item.sn:
                rows_by_key.setdefault((item.line, item.sn), []).append(item)
    for key, rows in rows_by_key.items():
        rows_by_key[key] = sorted(rows, key=lambda r: r.created_time)
    return rows_by_key


def _sample_rows(db_path: Path, *, line: str, include_inactive: bool, limit: int | None) -> list[sqlite3.Row]:
    where = ["sampling_rate IS NULL"]
    params: list[Any] = []
    if line:
        where.append("line = ?")
        params.append(line)
    if not include_inactive:
        where.append("is_active = 1")
    sql = (
        "SELECT id,line,sn,sample_id,group_name,channel_name,reference,time,"
        "tdms_storage_root,relative_path,tdms_path,is_active "
        "FROM samples WHERE "
        + " AND ".join(where)
        + " ORDER BY line,sn,sample_id,id"
    )
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return list(con.execute(sql, params))
    finally:
        con.close()


def _pick_manifest(row: sqlite3.Row, manifest: dict[tuple[str, str], list[ManifestRow]]) -> ManifestRow | None:
    candidates = list(manifest.get((_text(row["line"]), _text(row["sn"])), []))
    if not candidates:
        return None
    reference = _text(row["reference"])
    if reference:
        narrowed = [item for item in candidates if item.reference == reference]
        if narrowed:
            candidates = narrowed
    time = _text(row["time"])
    if time:
        narrowed = [item for item in candidates if item.time == time]
        if narrowed:
            candidates = narrowed
    return candidates[-1]


def _tdms_path(row: sqlite3.Row, manifest_row: ManifestRow | None, data_root: Path) -> Path | None:
    direct = _text(row["tdms_path"])
    if direct:
        return Path(direct).expanduser()
    storage = _text(row["tdms_storage_root"])
    relative = _text(row["relative_path"])
    if storage and relative:
        return data_root / storage / relative
    if manifest_row is None:
        return None
    if manifest_row.tdms_path:
        return Path(manifest_row.tdms_path).expanduser()
    if manifest_row.tdms_storage_root and manifest_row.relative_path:
        return data_root / manifest_row.tdms_storage_root / manifest_row.relative_path
    return None


def _infer_channel(row: sqlite3.Row, manifest_row: ManifestRow | None) -> tuple[str, str] | None:
    group_name = _text(row["group_name"])
    channel_name = _text(row["channel_name"])
    if group_name and channel_name:
        return group_name, channel_name

    line = _text(row["line"])
    sample_id = _text(row["sample_id"])
    sample_id_lower = sample_id.lower()
    if sample_id_lower.endswith("_up"):
        direction = "up"
    elif sample_id_lower.endswith("_down"):
        direction = "down"
    else:
        return None

    line_rule = LINE_RULES.get(line)
    if not line_rule:
        return None
    channels = line_rule.get("channels") or {}
    reference = _text(row["reference"]) or (_text(manifest_row.reference) if manifest_row else "")
    if "conditional" in channels:
        try:
            rules = SampleGenerator._select_conditional_rules(channels["conditional"], reference=reference)
        except Exception:
            return None
        if not rules:
            return None
        rule = rules[0]
    else:
        rule = channels
    return _text(rule.get(f"{direction}_group")), _text(rule.get("acc_channel"))


def _rate_from_channel(channel: Any) -> int | None:
    wf_increment = channel.properties.get("wf_increment")
    if not wf_increment:
        return None
    return int(round(1.0 / float(wf_increment)))


def _read_sampling_rate(tdms_path: Path, group_name: str = "", channel_name: str = "") -> int | None:
    if not tdms_path.is_file():
        return None
    with open_tdms(tdms_path, mode="read_metadata") as tdms:
        if group_name and channel_name and group_name in tdms and channel_name in tdms[group_name]:
            rate = _rate_from_channel(tdms[group_name][channel_name])
            if rate:
                return rate

        # Some historical TDMS metadata lacks wf_increment on the expected
        # analysis channel but still has it on another channel in the file.
        # Sampling rate is file acquisition metadata, so this is a valid
        # fallback for filling samples.sampling_rate.
        for group in tdms.groups():
            for channel in group.channels():
                rate = _rate_from_channel(channel)
                if rate:
                    return rate
    return None


def _update_database(db_path: Path, updates: list[tuple[int, int]]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "UPDATE samples SET sampling_rate = ?, updated_at = ? "
            "WHERE id = ? AND sampling_rate IS NULL",
            [(rate, now, sample_id) for sample_id, rate in updates],
        )
        con.commit()
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NULL samples.sampling_rate from TDMS metadata")
    parser.add_argument("--config", default="cfg/exp_e5_mel_lstm.yaml", help="YAML config used to resolve metadata paths")
    parser.add_argument("--database", default=None, help="Override label_records.db path")
    parser.add_argument("--manifest", default=None, help="Override tdms_manifest.csv path")
    parser.add_argument("--data-root", default=None, help="Override data_root")
    parser.add_argument("--line", default="", help="Only process one line, e.g. epump2")
    parser.add_argument("--include-inactive", action="store_true", help="Also process inactive samples")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--execute", action="store_true", help="Write updates to the database")
    args = parser.parse_args()

    data_root, db_path, manifest_path = _resolve_paths(args)
    if not db_path.is_file():
        raise FileNotFoundError(f"label_records.db not found: {db_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"tdms_manifest.csv not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    rows = _sample_rows(
        db_path,
        line=_text(args.line),
        include_inactive=bool(args.include_inactive),
        limit=args.limit,
    )

    stats: Counter[str] = Counter()
    updates: list[tuple[int, int]] = []
    rates: Counter[int] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}

    def remember(reason: str, row: sqlite3.Row, detail: str = "") -> None:
        stats[reason] += 1
        bucket = examples.setdefault(reason, [])
        if len(bucket) < 10:
            bucket.append({
                "id": int(row["id"]),
                "line": _text(row["line"]),
                "sn": _text(row["sn"]),
                "sample_id": _text(row["sample_id"]),
                "detail": detail,
            })

    for row in rows:
        manifest_row = _pick_manifest(row, manifest)
        tdms_path = _tdms_path(row, manifest_row, data_root)
        if tdms_path is None:
            remember("skip_no_manifest_or_path", row)
            continue
        channel = _infer_channel(row, manifest_row)
        group_name, channel_name = channel if channel else ("", "")
        try:
            sampling_rate = _read_sampling_rate(tdms_path, group_name, channel_name)
        except Exception as exc:
            remember("skip_read_error", row, f"{type(exc).__name__}: {exc}")
            continue
        if not sampling_rate or sampling_rate <= 0:
            remember("skip_no_sampling_rate", row, str(tdms_path))
            continue
        updates.append((int(row["id"]), int(sampling_rate)))
        rates[int(sampling_rate)] += 1
        stats["resolved"] += 1

    if args.execute and updates:
        _update_database(db_path, updates)
        stats["updated"] = len(updates)

    payload = {
        "database": str(db_path),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "execute": bool(args.execute),
        "candidate_rows": len(rows),
        "resolved_rows": len(updates),
        "sampling_rate_distribution": dict(sorted(rates.items())),
        "stats": dict(stats),
        "examples": examples,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        print("\nDRY-RUN: add --execute to write updates.")


if __name__ == "__main__":
    main()
