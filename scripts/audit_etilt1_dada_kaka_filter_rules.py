#!/usr/bin/env python3
"""Explain how training-label rules handle etilt1 dada/kaka samples."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.dataset.training_data.label_filter import _build_label_row_maps, _pick_training_label
from data_manager.label_database import load_label_dataframe, load_sample_dataframe


METADATA = Path("/Volumes/18555440521/fault/data_root/metadata")
REPORT = Path("reports/etilt1_dada_kaka_filter_decisions_20260608.csv")


def contains_dada_kaka(value: object) -> bool:
    text = str(value)
    return "哒哒" in text or "咔咔" in text


def main() -> None:
    history = load_label_dataframe(METADATA / "label_records.db").fillna("").astype(str)
    sample_index = load_sample_dataframe(METADATA / "label_records.db").fillna("").astype(str)
    sample_view = pd.read_csv(
        METADATA / "sample_view.csv",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    history = history[history["line"].eq("etilt1")].copy()
    dada = history[history["reason_name"].map(contains_dada_kaka)].copy()
    keys = sorted(set(zip(dada["line"], dada["sn"], dada["sample_id"])))
    by_triplet, by_pair = _build_label_row_maps(history)
    sample_index_keys = set(
        zip(sample_index["line"], sample_index["sn"], sample_index["sample_id"])
    )
    sample_view_keys = set(
        zip(sample_view["line"], sample_view["sn"], sample_view["sample_id"])
    )

    records: list[dict[str, object]] = []
    for key in keys:
        candidates = by_triplet.get(key) or by_pair.get((key[1], key[2]), [])
        chosen, decision = _pick_training_label(candidates)
        chosen_reason = (chosen or {}).get("reason_name", "")
        records.append(
            {
                "line": key[0],
                "sn": key[1],
                "sample_id": key[2],
                "history_dada_rows": sum(
                    contains_dada_kaka(row.get("reason_name", ""))
                    for row in candidates
                ),
                "all_history_rows": len(candidates),
                "decision": decision,
                "chosen_reason": chosen_reason,
                "kept_as_dada": bool(chosen and contains_dada_kaka(chosen_reason)),
                "in_sample_index": key in sample_index_keys,
                "in_sample_view": key in sample_view_keys,
                "sources": "|".join(
                    sorted(set(str(row.get("source", "")) for row in candidates))
                ),
                "all_reasons": "|".join(
                    sorted(set(str(row.get("reason_name", "")) for row in candidates))
                ),
            }
        )

    result = pd.DataFrame(records)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT, index=False, encoding="utf-8-sig")

    kept_non_dada = result["decision"].isin(["expert", "operator_consistent"]) & ~result[
        "kept_as_dada"
    ]
    print(f"dada_history_rows={len(dada)}")
    print(f"unique_sample_keys={len(result)}")
    print(f"unique_sns={result['sn'].nunique()}")
    print("\ndecision_sample_counts:")
    print(result["decision"].value_counts().to_string())
    print("\ndecision_unique_sn_counts:")
    print(result.groupby("decision")["sn"].nunique().sort_values(ascending=False).to_string())
    print(f"\nkept_as_dada={int(result['kept_as_dada'].sum())}")
    print(f"kept_as_dada_unique_sns={result.loc[result['kept_as_dada'], 'sn'].nunique()}")
    print(f"chosen_non_dada={int(kept_non_dada.sum())}")
    print(f"missing_sample_index={int((~result['in_sample_index']).sum())}")
    print(f"missing_sample_view={int((~result['in_sample_view']).sum())}")
    print("\nkept_dada:")
    print(
        result.loc[
            result["kept_as_dada"],
            ["sn", "sample_id", "decision", "sources", "all_reasons"],
        ].to_string(index=False)
    )
    print(f"\nreport={REPORT.resolve()}")


if __name__ == "__main__":
    main()
