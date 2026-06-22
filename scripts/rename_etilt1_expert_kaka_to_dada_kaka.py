#!/usr/bin/env python3
"""Rename etilt1 expert kaka labels to the standard dada/kaka name."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager.label_database import load_label_dataframe, replace_confirmed_labels  # noqa: E402

PATH = Path("/Volumes/18555440521/fault/data_root/metadata/label_records.db")


def main() -> None:
    frame = load_label_dataframe(PATH).fillna("").astype(str)
    mask = (
        frame["line"].eq("etilt1")
        & frame["reason_name"].eq("咔咔")
        & frame["source"].eq("expert")
        & frame["reason_key"].eq("dada")
        & frame["reason_id"].eq("109")
    )
    print(f"matched_rows={int(mask.sum())}")
    print(f"matched_samples={frame.loc[mask, ['line', 'sn', 'sample_id']].drop_duplicates().shape[0]}")
    print(f"matched_sns={frame.loc[mask, 'sn'].nunique()}")

    frame.loc[mask, "reason_name"] = "哒哒_咔咔"
    replace_confirmed_labels(PATH, frame.to_dict("records"))

    verify = load_label_dataframe(PATH).fillna("").astype(str)
    changed = (
        verify["line"].eq("etilt1")
        & verify["reason_name"].eq("哒哒_咔咔")
        & verify["source"].eq("expert")
        & verify["reason_key"].eq("dada")
        & verify["reason_id"].eq("109")
    )
    old_left = (
        verify["line"].eq("etilt1")
        & verify["reason_name"].eq("咔咔")
        & verify["reason_key"].eq("dada")
        & verify["reason_id"].eq("109")
    )
    print(f"changed_rows={int(changed.sum())}")
    print(f"old_kaka_rows_left={int(old_left.sum())}")


if __name__ == "__main__":
    main()
