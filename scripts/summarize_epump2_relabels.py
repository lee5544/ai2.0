#!/usr/bin/env python3
"""Aggregate epump2 relabel workbooks and compare them with label history."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import sys


RELABEL_ROOT = Path("/Volumes/18555440521/fault/data_root/factory_raw/epump2/再标记")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_manager.label_database import load_label_dataframe  # noqa: E402

HISTORY_CSV = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")
RULES_YAML = Path(__file__).resolve().parents[1] / "cfg" / "core" / "label_rules.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "epump2_relabel_summary_20260607"


def load_rules() -> tuple[dict, list[dict]]:
    rules = {"results": {}, "reasons": {}}
    section = ""
    current_key = ""
    in_aliases = False
    for line in RULES_YAML.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        text = line.strip()
        if indent == 0 and text.endswith(":"):
            section = text[:-1]
            current_key = ""
            in_aliases = False
        elif section in rules and indent == 2 and text.endswith(":"):
            current_key = text[:-1]
            rules[section][current_key] = {"alias": []}
            in_aliases = False
        elif section in rules and current_key and indent == 4 and ":" in text:
            field, raw_value = text.split(":", 1)
            raw_value = raw_value.strip()
            in_aliases = field == "alias"
            if field == "id":
                rules[section][current_key][field] = int(raw_value)
            elif raw_value == "[]":
                rules[section][current_key][field] = []
            elif raw_value:
                rules[section][current_key][field] = raw_value.strip("'\"")
        elif section in rules and current_key and in_aliases and indent == 4 and text.startswith("- "):
            rules[section][current_key]["alias"].append(text[2:].strip("'\""))
    reason_lookup = {}
    rule_rows = []
    for key, value in rules["reasons"].items():
        reason_lookup[key] = value
        rule_rows.append(
            {
                "rule_type": "reason",
                "key": key,
                "id": value["id"],
                "name": value["name"],
                "parent": value["parent"],
                "aliases": " | ".join(value.get("alias", [])),
            }
        )
    for key, value in rules["results"].items():
        rule_rows.append(
            {
                "rule_type": "result",
                "key": key,
                "id": value["id"],
                "name": value["name"],
                "parent": "",
                "aliases": " | ".join(value.get("alias", [])),
            }
        )
    return reason_lookup, rule_rows


def map_reason(status: str, raw_type: str) -> str:
    status = str(status).strip().upper()
    raw = str(raw_type).strip().lower()
    if status == "OK":
        return "clean_normal"
    if status not in {"NOK", "NG"}:
        return ""
    raw = re.sub(r"^(轻微|中等|强烈|轻度|重度)[-_—–\s]*", "", raw)
    if "传感器" in raw or "sensor" in raw:
        return "sensor_error"
    if "秒表" in raw or "tick" in raw:
        return "tick_tock"
    if "摩擦acc" in raw or ("摩擦" in raw and "acc" in raw):
        return "friction_acc"
    if "摩擦" in raw:
        return "friction"
    if "杂音" in raw or "噪声" in raw or "noise" in raw:
        return "noise"
    if "咬齿" in raw or "齿轮" in raw or "gear" in raw:
        return "gear_chatter"
    if "震颤" in raw or "抖动" in raw or "颤振" in raw:
        return "chatter"
    if "马达" in raw:
        return "mada"
    if "哒哒" in raw or "咔咔" in raw:
        return "dada"
    if "边界" in raw or "不确定" in raw or "无法判断" in raw:
        return "boundary"
    return "other"


def result_key(status: str) -> str:
    status = str(status).strip().upper()
    if status == "OK":
        return "ok"
    if status in {"NOK", "NG"}:
        return "nok"
    if status in {"BOUNDARY", "边界", "不确定", "待确认"}:
        return "boundary"
    return ""


def value_counts_rows(df: pd.DataFrame, scope: str, field: str) -> list[dict]:
    counts = df[field].fillna("").replace("", "空白").value_counts(dropna=False)
    total = int(counts.sum())
    return [
        {
            "scope": scope,
            "field": field,
            "category": str(category),
            "count": int(count),
            "percentage": count / total if total else 0,
        }
        for category, count in counts.items()
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reason_lookup, rule_rows = load_rules()
    workbook_rows = []
    sample_rows = []

    for path in sorted(RELABEL_ROOT.rglob("label.xlsx")):
        batch = path.parent.name
        excel = pd.ExcelFile(path)
        for sheet in excel.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
            for row_number, row in frame.iterrows():
                base = {
                    "batch": batch,
                    "source_file": str(path),
                    "round": str(sheet),
                    "row_number": int(row_number + 2),
                    **{column: str(row.get(column, "")).strip() for column in frame.columns},
                }
                workbook_rows.append(base)
                if not base["result"] or str(sheet) not in {"1", "2"}:
                    continue
                for direction in ("up", "down"):
                    status = base[direction]
                    raw_type = base[f"{direction}_type"]
                    reason = map_reason(status, raw_type)
                    reason_rule = reason_lookup.get(reason, {})
                    sample_rows.append(
                        {
                            "batch": batch,
                            "source_file": str(path),
                            "round": str(sheet),
                            "row_number": int(row_number + 2),
                            "reference": base["reference"],
                            "sn": base["sn"].upper(),
                            "sample_id": f"{base['sn'].upper()}_{direction}",
                            "direction": direction,
                            "raw_status": status,
                            "raw_type": raw_type,
                            "result_key": result_key(status),
                            "reason_key": reason,
                            "reason_id": reason_rule.get("id", ""),
                            "reason_name": reason_rule.get("name", ""),
                            "operator_id": base["operator_id"],
                            "label_time": base["label_time"],
                        }
                    )

    workbook_df = pd.DataFrame(workbook_rows)
    samples_df = pd.DataFrame(sample_rows)
    final_df = samples_df[samples_df["round"] == "2"].copy()
    valid_sn = set(samples_df["sn"])

    history_df = load_label_dataframe(HISTORY_CSV).fillna("").astype(str)
    history_df["sn"] = history_df["sn"].str.upper()
    history_df = history_df[(history_df["line"] == "epump2") & history_df["sn"].isin(valid_sn)].copy()

    if not history_df.empty:
        history_df["timestamp_sort"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
        history_df = history_df.sort_values(["sample_id", "timestamp_sort", "source"])
        history_summary = (
            history_df.groupby("sample_id", as_index=False)
            .agg(
                history_record_count=("sample_id", "size"),
                history_sources=("source", lambda x: " | ".join(sorted(set(x.dropna())))),
                history_results=("result_key", lambda x: " | ".join(sorted(set(x.dropna())))),
                history_reasons=("reason_key", lambda x: " | ".join(sorted(set(x.dropna())))),
            )
        )
        latest_history = history_df.groupby("sample_id", as_index=False).tail(1).rename(
            columns={
                "timestamp": "history_latest_timestamp",
                "source": "history_latest_source",
                "result_key": "history_latest_result_key",
                "reason_key": "history_latest_reason_key",
                "reason_name": "history_latest_reason_name",
                "label_version": "history_latest_label_version",
            }
        )
        latest_history = latest_history[
            [
                "sample_id",
                "history_latest_timestamp",
                "history_latest_source",
                "history_latest_result_key",
                "history_latest_reason_key",
                "history_latest_reason_name",
                "history_latest_label_version",
            ]
        ]
        comparison_df = final_df.merge(history_summary, on="sample_id", how="left").merge(
            latest_history, on="sample_id", how="left"
        )
    else:
        comparison_df = final_df.copy()
        for column in [
            "history_record_count",
            "history_sources",
            "history_results",
            "history_reasons",
            "history_latest_timestamp",
            "history_latest_source",
            "history_latest_result_key",
            "history_latest_reason_key",
            "history_latest_reason_name",
            "history_latest_label_version",
        ]:
            comparison_df[column] = ""

    comparison_df["has_history"] = comparison_df["history_record_count"].notna()
    comparison_df["result_same_as_history_latest"] = (
        comparison_df["result_key"] == comparison_df["history_latest_result_key"]
    ) & comparison_df["has_history"]
    comparison_df["reason_same_as_history_latest"] = (
        comparison_df["reason_key"] == comparison_df["history_latest_reason_key"]
    ) & comparison_df["has_history"]
    comparison_df["comparison_status"] = "无历史"
    comparison_df.loc[
        comparison_df["has_history"] & comparison_df["reason_same_as_history_latest"], "comparison_status"
    ] = "结果及原因一致"
    comparison_df.loc[
        comparison_df["has_history"]
        & comparison_df["result_same_as_history_latest"]
        & ~comparison_df["reason_same_as_history_latest"],
        "comparison_status",
    ] = "结果一致/原因不同"
    comparison_df.loc[
        comparison_df["has_history"] & ~comparison_df["result_same_as_history_latest"], "comparison_status"
    ] = "结果不同"

    round1 = samples_df[samples_df["round"] == "1"][
        ["sample_id", "result_key", "reason_key"]
    ].rename(columns={"result_key": "round1_result_key", "reason_key": "round1_reason_key"})
    comparison_df = comparison_df.merge(round1, on="sample_id", how="left")
    comparison_df["rounds_result_same"] = comparison_df["result_key"] == comparison_df["round1_result_key"]
    comparison_df["rounds_reason_same"] = comparison_df["reason_key"] == comparison_df["round1_reason_key"]

    sn_rows = []
    for sn, group in comparison_df.groupby("sn", sort=True):
        by_dir = {row["direction"]: row for _, row in group.iterrows()}
        up = by_dir.get("up", {})
        down = by_dir.get("down", {})
        sn_rows.append(
            {
                "sn": sn,
                "batch": group["batch"].iloc[0],
                "reference": group["reference"].iloc[0],
                "final_result": "nok" if "nok" in set(group["result_key"]) else "ok",
                "up_result": up.get("result_key", ""),
                "up_reason": up.get("reason_key", ""),
                "down_result": down.get("result_key", ""),
                "down_reason": down.get("reason_key", ""),
                "rounds_both_result_same": bool(group["rounds_result_same"].all()),
                "rounds_both_reason_same": bool(group["rounds_reason_same"].all()),
                "history_sample_count": int(group["has_history"].sum()),
                "history_result_same_count": int(group["result_same_as_history_latest"].sum()),
                "history_reason_same_count": int(group["reason_same_as_history_latest"].sum()),
                "history_comparison": " | ".join(
                    f"{row['direction']}:{row['comparison_status']}" for _, row in group.sort_values("direction").iterrows()
                ),
            }
        )
    sn_df = pd.DataFrame(sn_rows)

    distribution_rows = []
    for round_name in ["1", "2"]:
        round_samples = samples_df[samples_df["round"] == round_name]
        distribution_rows += value_counts_rows(round_samples, f"轮次{round_name}-方向样本", "result_key")
        distribution_rows += value_counts_rows(round_samples, f"轮次{round_name}-方向样本", "reason_key")
    distribution_rows += value_counts_rows(sn_df, "最终-按SN", "final_result")
    distribution_rows += value_counts_rows(comparison_df, "最终-历史对比", "comparison_status")
    distribution_df = pd.DataFrame(distribution_rows)

    unmapped_df = samples_df[
        (samples_df["raw_status"].str.upper().isin(["NOK", "NG"])) & (samples_df["reason_key"] == "other")
    ][["raw_type"]].value_counts().reset_index(name="count")

    summary = {
        "label_xlsx_count": len(list(RELABEL_ROOT.rglob("label.xlsx"))),
        "all_workbook_rows": int(len(workbook_df)),
        "valid_label_rows": int(len(samples_df) / 2),
        "unique_sn": int(samples_df["sn"].nunique()),
        "final_sample_count": int(len(final_df)),
        "history_matched_samples": int(comparison_df["has_history"].sum()),
        "history_unmatched_samples": int((~comparison_df["has_history"]).sum()),
        "latest_history_result_same": int(comparison_df["result_same_as_history_latest"].sum()),
        "latest_history_reason_same": int(comparison_df["reason_same_as_history_latest"].sum()),
        "rounds_result_same": int(comparison_df["rounds_result_same"].sum()),
        "rounds_reason_same": int(comparison_df["rounds_reason_same"].sum()),
    }

    for name, frame in {
        "all_workbook_rows": workbook_df,
        "sample_labels": samples_df,
        "final_history_comparison": comparison_df,
        "sn_summary": sn_df,
        "distribution": distribution_df,
        "matched_history": history_df,
        "unmapped_types": unmapped_df,
        "rules": pd.DataFrame(rule_rows),
    }.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_data = {
        "summary": summary,
        "distribution": distribution_df.fillna("").to_dict("records"),
        "sn_summary": sn_df.fillna("").to_dict("records"),
        "final_history_comparison": comparison_df.fillna("").to_dict("records"),
        "sample_labels": samples_df.fillna("").to_dict("records"),
        "all_workbook_rows": workbook_df.fillna("").to_dict("records"),
        "unmapped_types": unmapped_df.fillna("").to_dict("records"),
        "rules": pd.DataFrame(rule_rows).fillna("").to_dict("records"),
    }
    (OUTPUT_DIR / "report_data.json").write_text(
        json.dumps(report_data, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
